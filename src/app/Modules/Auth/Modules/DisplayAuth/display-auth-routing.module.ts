import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {DisplayAuthComponent} from "./display-auth/display-auth.component";

const routes: Routes = [
    {
        path: '',
        component: DisplayAuthComponent,
    }
];

@NgModule({
    imports: [RouterModule.forChild(routes)],
    exports: [RouterModule]
})
export class DisplayAuthRoutingModule {
}
