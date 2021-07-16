!This module contains several module variables (see comments below)
!and two subroutines:
!simulate_jacobi: Uses jacobi iteration to compute solution
!to contamination model problem
!simulate: To be completed. Use over-step iteration method to
!simulate contamination model

module bmodel
    implicit none
    integer :: bm_kmax=10000 !max number of iterations
    real(kind=8) :: bm_tol=0.00000001d0 !convergence criterion
    real(kind=8), allocatable, dimension(:) :: deltac !|max change in C| each iteration
    real(kind=8) :: bm_g=1.d0,bm_kbc=1.d0 !death rate, r=1 boundary parameter
    real(kind=8) :: bm_s0=1.d0,bm_r0=2.d0,bm_t0=1.5d0 !source parameters

contains
!-----------------------------------------------------
!Solve 2-d contaminant spread problem with Jacobi iteration
subroutine simulate_jacobi(n,C)
    !input  n: number of grid point (n+2 x n+2) grid
    !output C: final concentration field
    !       deltac(k): max(|C^k - C^k-1|)
    !A number of module variables can be set in advance.

    integer, intent(in) :: n
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    integer :: i1,j1,k1
    real(kind=8) :: pi,del,del2f
    real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,Cnew,fac,fac2,facp,facm

    if (allocated(deltac)) then
      deallocate(deltac)
    end if
    allocate(deltac(bm_kmax))

    pi = acos(-1.d0)

    !grid--------------
    del = pi/dble(n+1)
    del2f = 0.25d0*(del**2)


    do i1=0,n+1
        r(i1,:) = 1.d0+i1*del
    end do

    do j1=0,n+1
        t(:,j1) = j1*del
    end do
    !-------------------

    !Update equation parameters------
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*k^2 + 2 + 2/r^2)
    facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
    facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
    fac2 = fac*rinv2
    !-----------------

    !set initial condition/boundary conditions
    C = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi

    !set source function, Sdel2 = S*del^2*fac
    Sdel2 = bm_s0*exp(-20.d0*((r-bm_r0)**2+(t-bm_t0)**2))*(del**2)*fac

    !Jacobi iteration
    do k1=1,bm_kmax
        Cnew(1:n,1:n) = Sdel2(1:n,1:n) + C(2:n+1,1:n)*facp(1:n,1:n) + C(0:n-1,1:n)*facm(1:n,1:n) + &
                                         (C(1:n,0:n-1) + C(1:n,2:n+1))*fac2(1:n,1:n)
        deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error
        C(1:n,1:n)=Cnew(1:n,1:n)    !update variable
        if (deltac(k1)<bm_tol) exit !check convergence criterion
        if (mod(k1,1000)==0) print *, k1,deltac(k1)
    end do

    print *, 'k,error=',k1,deltaC(min(k1,bm_kmax))

end subroutine simulate_jacobi
!-----------------------------------------------------

!Solve 2-d contaminant spread problem with over-step iteration method
subroutine simulate(n,C)
    !input  n: number of grid point (n+2 x n+2) grid
    !output C: final concentration field
    !       deltac(k): max(|C^k - C^k-1|)
    !A number of module variables can be set in advance.

    integer, intent(in) :: n
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    !Add other variables as needed
    integer :: i1,j1,k1
    real(kind=8) :: pi,del,del2f
    real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,Cnew,fac,fac2,facp,facm

    !Set size of deltac
    if (allocated(deltac)) then
      deallocate(deltac)
    end if
    allocate(deltac(bm_kmax))

    !Set up pi
    pi = acos(-1.d0)

    !grid--------------
    del = pi/dble(n+1)
    del2f = 0.25d0*(del**2)

    !Set up numerical parameters, r
    do i1=0,n+1
        r(i1,:) = 1.d0+i1*del
    end do

    !Set numerical parameters, t
    do j1=0,n+1
        t(:,j1) = j1*del
    end do
    !-------------------

    !Update equation parameters------
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*k^2 + 2 + 2/r^2)
    facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
    facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
    fac2 = fac*rinv2
    !-----------------

    !set initial condition/boundary conditions
    C = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi

    !set source function, Sdel2 = S*del^2*fac
    Sdel2 = bm_s0*exp(-20.d0*((r-bm_r0)**2+(t-bm_t0)**2))*(del**2)*fac

    !Initialize Cnew before the loop starts
    Cnew = C
    !Intiialize deltac before the loop starts
    deltac=0.d0

    !OSI iteration
    do k1=1,bm_kmax
      !Compute cnew, progress through matrix element-by-element one row at a time
      do i1=1,n
        do j1=1,n
        Cnew(i1,j1)=0.5d0*(-C(i1,j1)+3.d0*(facm(i1,j1)*Cnew(i1-1,j1)+&
        fac2(i1,j1)*Cnew(i1,j1-1)+fac2(i1,j1)*C(i1,j1+1)+&
        facp(i1,j1)*C(i1+1,j1)+Sdel2(i1,j1)))
        end do
      end do

      !Compute delta c
      deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error

      !Update C before next iteration starts
      C(1:n,1:n)=Cnew(1:n,1:n)    !update variable

      !Iterations terminate if max change falls below tolerance
      if (deltac(k1)<bm_tol) exit !check convergence criterion

      !Print statement at every 1000 iterations
      if (mod(k1,1000)==0) print *, k1,deltac(k1)
    end do

!Print final iteration and error
print *, 'k,error=',k1,deltaC(min(k1,bm_kmax))

end subroutine simulate
!--------------------------------------------------------------------------------
!This subroutine is a modified verison of the simulate subroutine
!The subroutine has the modified contamination on or r=1 and has more input variables
!The new input variable is tstar which can be varied and is used for anlyze in part (4)
subroutine simulate_antibac(bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,n,tstar,C)
  integer, intent(in) :: n
  real(kind=8), intent(in) :: bm_g,bm_r0,bm_kbc,bm_s0,bm_t0,tstar
  real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
  !Add other variables as needed
  integer :: i1,j1,k1
  real(kind=8) :: pi,del,del2f
  real(kind=8), dimension(0:n+1,0:n+1) :: r,rinv2,t,Sdel2,Cnew,fac,fac2,facp,facm

  if (allocated(deltac)) then
    deallocate(deltac)
  end if
  allocate(deltac(bm_kmax))

  pi = acos(-1.d0)

  !grid--------------
  del = pi/dble(n+1)
  del2f = 0.25d0*(del**2)

  do i1=0,n+1
      r(i1,:) = 1.d0+i1*del
  end do

  do j1=0,n+1
      t(:,j1) = j1*del
  end do
  !-------------------

  !Update equation parameters------
  rinv2 = 1.d0/(r**2)
  fac = 1.d0/(2.d0+2.d0*rinv2+del*del*bm_g) !1/(del^2*k^2 + 2 + 2/r^2)
  facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
  facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
  fac2 = fac*rinv2
  !-----------------

  !set initial condition/boundary conditions
  C = (sin(bm_kbc*t)**2)*(pi+1.d0-r)/pi
  !Modified boundary condition
  C(0,:) = exp(-10.d0*(t(0,:)-tstar)**2)*(sin(bm_kbc*t(0,:))**2)


  !set source function, Sdel2 = S*del^2*fac
  Sdel2 = bm_s0*exp(-20.d0*((r-bm_r0)**2+(t-bm_t0)**2))*(del**2)*fac

  !Initialize Cnew to use in iterations
  Cnew = C
  !Initialize deltac to use in iterations
  deltac=0.d0

  !OSI iteration
  do k1=1,bm_kmax
    do i1=1,n
      do j1=1,n
      Cnew(i1,j1)=0.5d0*(-C(i1,j1)+3.d0*(facm(i1,j1)*Cnew(i1-1,j1)+&
      fac2(i1,j1)*Cnew(i1,j1-1)+fac2(i1,j1)*C(i1,j1+1)+&
      facp(i1,j1)*C(i1+1,j1)+Sdel2(i1,j1)))
      end do
    end do

    deltac(k1) = maxval(abs(Cnew(1:n,1:n)-C(1:n,1:n))) !compute relative error
    C(1:n,1:n)=Cnew(1:n,1:n)    !update variable
    if (deltac(k1)<bm_tol) exit !check convergence criterion
    if (mod(k1,1000)==0) print *, k1,deltac(k1)
  end do

  print *, 'k,error=',k1,deltaC(min(k1,bm_kmax))

end subroutine simulate_antibac


end module bmodel
